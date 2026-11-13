# Temporary File Policy

This is a shared policy used by all skills that create files during their workflow.
The goal is to avoid leaving behind clutter and to give the user control over where
artifacts are stored.

## Session-Level Location

At the start of each session, before the first file-creation operation, ask the user
where temporary files should be stored. Once the user specifies a location, reuse it
for all temporary files in that session without asking again. If the user has not yet
specified a location when a skill needs to create files, ask before proceeding.

## Before Creating Files

Before writing any files to disk (scripts, images, intermediate artifacts, build
directories, venvs, etc.):

1. **Ask the user where to put them** (unless already established for this session — see
   Session-Level Location above). Never assume a location. Offer a sensible default
   but let the user choose. For example:
   - "I need to create a `generate_diagrams.py` script and an `img/` directory.
     Where should I put these? Default: alongside the source file."
   - "This build needs a build directory. Where should I create it?
     Default: `<project-root>/build`"

2. **Distinguish persistent vs temporary artifacts.** Tell the user which files are
   intended to be kept (e.g., documentation, diagrams, test files) and which are
   intermediate/throwaway (e.g., build caches, HTML intermediates, venvs).

3. **Record what you create.** Keep a mental list of all files and directories created
   during the session so you can offer cleanup at the end.

## During the Session

- **MUST NOT install packages on the bare machine.** This is a shared system — installing
  packages globally affects other users and can break their environments. This rule is
  non-negotiable and applies to all skills without exception.
  - **Python packages**: always create a venv first (`python3 -m venv .venv-<purpose>`)
    and install inside it. Name the venv descriptively (e.g., `.venv-pdf`, `.venv-profiling`).
  - **System packages** (`apt`, `yum`, `pip install --user`): never on the host. If a
    system package is needed, install it inside a container (Docker or enroot).
  - **npm, cargo, or other language packages**: same rule — use a container or isolated
    environment, never the bare machine.
  - **ROCm tools or compilers**: these are managed by the system admin. Never install
    or upgrade them. Use what is available in `/opt/rocm` or inside a container.
- **MUST NOT create files outside the project directory** without explicit user permission.
- **Use descriptive names** for temporary files so the user can identify them later.
  Avoid generic names like `temp.txt` or `output.dat`.
- **Prefer creating files in a single, contained location** (e.g., one temp directory)
  rather than scattering files across multiple directories.

## Shared System Etiquette

This machine is shared. Be a good citizen.

### Disk Space Awareness
- **Before creating large artifacts** (build directories, trace JSONs, squashfs images,
  large logs), check available disk space. Use platform-appropriate commands:
  - Linux: `df -h /tmp && df -h .`
  - Windows (Git Bash): `df -h . 2>/dev/null || wmic logicaldisk get size,freespace,caption 2>/dev/null`
- **Warn the user** if available space is below 20% or if the artifact being created
  is expected to be large (>1GB).
- **Monitor cumulative size** of files created during the session. If total created
  files exceed 10GB, proactively inform the user.

### Stale File Detection
- **At the start of a session**, check for leftover artifacts from previous sessions
  in the project directory (`.venv-*`, `build-*` directories):
  - Linux: `find . -maxdepth 1 -name ".venv-*" -mtime +7 -type d 2>/dev/null`
  - Windows (Git Bash): `find . -maxdepth 1 -name ".venv-*" -type d 2>/dev/null` (check dates manually)
  - Cross-platform alternative: use `ls -lt .venv-* build-* 2>/dev/null` and check dates
- **If stale files are found**, offer to clean them up with context (size, age).
- **Never delete stale files without asking.** The user may have ongoing work.

### Cleanup Reminders
- If a session creates temporary files and the user ends the conversation without
  cleaning up, remind them:
  - "Before we wrap up: I created these temporary files during this session that
    can be cleaned up: [list]. Want me to remove them?"

## After the Task Completes

When the skill's primary task is finished:

1. **List all files created during the session.** Group them by category:
   - **Keep**: Files the user likely wants to retain (docs, diagrams, test files)
   - **Clean up**: Intermediate artifacts the user likely doesn't need (build caches,
     HTML intermediates, temp scripts, venvs created for one-time use)

2. **Ask the user what to clean up.** Don't delete anything without asking. For example:
   - "I created the following temporary files during this session:
     - `.venv-pdf/` (Python venv for PDF generation)
     - `img/component_architecture.html` (intermediate HTML)
     Want me to clean these up?"

3. **If the user says yes, delete the temporary files** and confirm what was removed.

4. **If the user says no or doesn't respond, leave everything in place.** Never
   silently clean up files the user might want to inspect.

## Exceptions

- **Build directories** managed by cmake/ninja are long-lived and should not be
  offered for cleanup after each build. Only offer cleanup if the user explicitly
  asks to clean the build.
- **Git-ignored files** (files matched by `.gitignore`) are less critical to clean up
  but should still be mentioned.
- **Docker containers** are not files but should follow the same principle: ask before
  creating, offer to stop/remove when done.

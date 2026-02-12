---
on:
  schedule:
    - cron: "0 9 * * 1,3,5"  # M,W,F at 9am UTC
permissions:
  contents: read
engine:
  id: claude
  model: claude-sonnet-4
safe-outputs:
  create-pull-request:
---

# Hipblaslt Documentation Updater

The task is to provide excellent documentation for projects/hipblaslt/tensilelite/Tensile/

At that directory and each sub-directory below, the desired state is to have a docs/ directory that
documents code that lives at that level of the directory hiearchy.

This agent will have one of these tasks:

- a docs/ directory exists, but it's not good enough/complete/correct, and so work must be done there
- a docs/ directory does not exist, but it should, so one will be created and some documentation started
- a docs/ directory exists at a certain location, but it needs refactoring based on code changes or
  the agent decides that some re-work could be done to enhance clarity, quality etc.

Documentation needs to be added and reviewed by human authors, and so a "chunk" of work done by this agent
in a single pull request should be no more than 100 new lines of text (deleting lines of documentation does
not count towards this total).

This agent will work in a bread-first-search style of adding documentation (adding docs/ directories
across the span directories in the current directory, rather than pushing "down" as the docs are filled out).

If no docs/ directory exists the current docs/ directory looks good, then proceed to the "next" directory,
create a docs/ directory there, and proceed to start the work (see "What to do" section next).

It is preferable to create new documentation in a new docs/ directory, rather than continue to wordsmith
existing documentation and refine. However, if you detect that files have changed and the documentation is stale,
that is the highest priority action to take.

## Creating a new documentation file

Documentation files are in Markdown, ending with the extension ".md". They are named with camel case, e.g.
"TensileCodegenConcepts.md".

The goal is not to create a documentation file for every source file in a directory, but rather to
provide a nice overview of the files in the directory with a total of 3-5 total markdown files that
discuss the code in the directory. A reasonable pattern is probably

- <DIR NAME>Overview.md - a markdown file that gives a high level view of the code in the directory this docs/
  directory
- <Concept1>.md - this might be a file that shares the same root name as an important source file at this level,
  or it may be some conceptual topic that is important to understand for the execution flow covered the code here.
- <SourceFileRoot1>.md - a drill down on a particulary important file in the containing directory.
- <SourceFileRoot2>.md - a drill down on a second file in the containing directory.
- <SourceFileRoot3>.md - a drill down on a theird file in the containing directory.

## A "TODO" documentation file
You may find a a file, e.g. "Foo.md", in a docs/ directory with just the contents:

TODO - <instructions here>

If you come across such a file, that becomes the MOST IMPORTANT task for this pull request. Look at the filename
and the instructions and decide what to do to create good documentation for that file.


When working in a docs/ directory, scan source files for new or changed code that doesn't match the existing docs.
Updating existing documentation that is wrong or outdated is more important than adding new documentation.
Adding new documentation is more important than refining existing correct documentation.

## Style guidelines
- Use present tense
- Keep explanations concise

## The definition of "Done"
- Follow these guidelines above, continuing to do work until you approach that ~100 line limit of new content
  OR you observe that mostly the documentation looks good, is present everywhere it needs to be, and nothing
  more needs to be done.

## When Done
- Create a pull request with all documentation changes

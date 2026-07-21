# Using CodeQL

## Installation

`install_codeql` called by the docker files to install the CodeQL CLI and external queries.  If invoking this script manually, be sure to run `export PATH=$PATH:/opt/codeql/codeql`.

## Setup

`setup_codeql` sets up the directories and externally authored queries.

## Create Database

`create_database` builds the cpp project and creates the CodeQL database.

## Analyse / Run Queries

`analyze_database` runs all queries on the CodeQL database and outputs the result to `build/codeql.sarif` and `build/codeql.html`. The errors/warnings/recommendations can be visualised with a [SARIF viewer](https://sarifweb.azurewebsites.net/#Viewers).

## Creating New Queries

New queries should be stored in `<language>/src/`. Libraries that may be included in one or more queries should be stored in `<language>/lib/` (libraries cannot contain queries).

### Run A Single Query

`run_single_query <path/to/query.ql>` runs a singular query.
For example, `codeql/run_single_query codeql/cpp/src/ToStringNaming.ql`.

### Create Tests for Query

`codeql/inflate_new_test_dir.py -p codeql/<language>/test/ <NewQueryName>` to setup a directory with the files required for testing.
For example `codeql/inflate_new_test_dir.py -p codeql/cpp/test/ ToStringNaming`.

### Run Tests

`codeql/run_tests` to run tests for the queries.

## Useful Docummentation

[CodeQL for C and C++](https://codeql.github.com/docs/codeql-language-guides/codeql-for-cpp/)
[QL Language Reference](https://codeql.github.com/docs/ql-language-reference/)

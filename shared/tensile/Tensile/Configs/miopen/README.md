Intro:

Tensile YAML files are composed of 5 components (Header, (Problem) Type, Solution(s), Problems, and Footer.
The files here are partitioned into modular sections that allow these sections to be maintained.
and documented separately, and combined in very flexible manner using an easy Makefile-based editing system.
  - Problem sizes (DeepSpeech, ResNet50, etc) can be kept in a single file and then merged as-needed into YAML files
    for different problem type (Single, Half, and Mixed).
  - Different solutions (a quick tune or an expensive explore) can be easily selected.

A sample Makefile is provides to combine the YAML file.  Makefile provides simple Variable support to eliminate
replication and provide some composability.
On some shells (zsh), autocompletion identifies valid targets in the makefile and provides a convenient
commandline shortcut for specifying targets.


The components all use the extension 'yml' to distinguish them from Tensile inputs (yaml).
These are the 5 components that are combined into a Tensile YAML file:
- Header :
  GlobalParameters such as DataInitTypes, SolutionSelectionAlg, SleepPercent, CMakeBuildType, etc.
  Example: boiler/header.yml

- Type  :
  The problem type such as (transpose ops, dataType (half,float,double), mixed, UseBeta, etc
  Example: types/sgemm_nn.yml

- Solution(s) :
  Multiple parameters describing the solutions.  A Tensile YAML file can contain multiple solution sections, each
  followed by one or more problems.  Solutions can vary significantly in how many solutions they create and thus
  how long Tensile requires to generate kernels and explore the design space.  For example:
  - Quick solutions create a small set of solutions that can be quickly explored.
  - Explore solutions create a large number of solutiosn that take a few hours to explore.

  - Some solutions can target different matrix configs.  For example:
    - 'large' targets large,square,friendly matrix configs where large tiles sizes and prefetching are winners.
    - 'skinny' targets smaller configs where register consumption and smaller tile sizes are winners.
    - other configs may be appropriate for powers-of-two or other unusual configs.

  - Example: solutions/nn/sgemm_quick.yml



- Problem(s):
  - Problems define Exact or Ranges to explore.
  - Example: problems/nn/deepbench_large.yml

- Footer:
  Define library logic that should be used.

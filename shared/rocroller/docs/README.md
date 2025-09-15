

# HTML Documentation Instructions

## Doxygen

Within the directory `docs/src/`, the file `Doxyfile` controls what is produced on the Doxygen part of the website. Likewise, the ```ReadTheDocs``` themed Sphinx webpage is produced via the `conf.py` file. Both pages are build using the the build system created by the CMake configuration step. The Doxygen site is built within the Sphinx directory in the documentation build folder. CMake commands also control how the two websites are stitched together.

## Adding documentation

Documents can be in either Markdown or reStructuredText File format. Both can be included into the generated website documention. There are differences in their capabilites that this README will not address.

### Adding documentation into the `docs` folder

Pages that do not fit in any particular source folder can be placed in the `docs/src` folder. Example of these may be overall design documentation or auxiliary behaviors or tools.

Images can be contained within the documents, however artifacts (the image files themselves) need to be added to the `docs/src/images` folder. Any other artifacts should be dropped into a new `docs/src/<artifact_label>` folder.

After adding the documentation files, the file `docs/src/index.rst` needs to be updated. `index.rst` contains the table of contents for the entire HTML documentation. The file name must be added to the content in the order the document requires. One note of care, the name of the entry in the table of contents must be exact to name of the file being included, minus the suffix, and ***must*** match the case. For example if the file `FooBar.md` is to be included, the entry would read `FooBar` in the table of contents.


### Adding a document from another folder

Sometimes it makes sense to have the documentation file resident in the source directory it is referencing. In that case, `CMakeLists.txt` needs to be edited to pull in the file to the `docs/<build_directory>` and deleted when a developer executes a `make clean` command.

In the section labeled `# Aux docs` in `CMakeLists.txt` add a custom target in this format:

```
add_custom_target(<copyover_some_readme>
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../<path_to_doc>/<doc_name> ${DOCS_BUILD_DIR}/src/<new_file_name>
    COMMENT "Copying over <doc_name> to docs/build/src folder as <new_file_name>."
    DEPENDS copysrc_to_builddir
)
```

Rename the `< >` bracketed sections to the names and paths needed. Then the custom target under the label `# Prebuild Sphinx` needs editing to add the created custom target's name:

```
# Prebuild Sphinx
add_custom_target(prebuild_sphinx
    DEPENDS
        copysrc_to_builddir
        clip_designreadme
        copyover_mainreadme
        copyover_dockerreadme
        <copyover_some_readme>
        build_doxygen
)
```

The order here does not matter. Finally, don't forget to update the `docs/src/index.rst` file with the `<new_file_name>` without the file type suffix.

`CMakeLists.txt` can also be used to modify the file before the website is built. There are multiple examples of document editing in that file.


### Mermaid Charts

Sphinx is able to render Mermaid charts such as:

```{mermaid}
graph TD

  UA(["User(A)"])
  SDA0(["SubDimension(A; 0)"])
  SDA1(["SubDimension(A; 1)"])
  MTAN0(["MacroTileNumber(A; 0)"])
  MTAI0(["MacroTileIndex(A; 0)"])
  MTAN1(["MacroTileNumber(A; 1)"])
  MTAI1(["MacroTileIndex(A; 1)"])

  FK(["K Loop Iterations (512)"])

  UA -- Split --> SDA0 & SDA1
  SDA0 -- Tile --> MTAN0 & MTAI0
  SDA1 -- Tile --> MTAN1 & MTAI1
  MTAN1 -- PassThrough --> FK

  style MTAN0 stroke:#3f3
  style FK stroke:#3f3
```

However, GitHub's style differs slightly from the Sphinx plugin. GitHub expects `mermaid::` while Sphinx expects `{mermaid}`.
This means a sed command needs to be run to find and replace the string. Here is an example:

```
add_custom_target(<copyover_some_readme>
    COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/../<path_to_doc>/<doc_name> ${DOCS_BUILD_DIR}/src/<new_file_name>
    COMMAND sed -i 's;mermaid::;{mermaid};g' ${DOCS_BUILD_DIR}/src/<new_file_name>
    COMMENT "Copying over <doc_name> to docs/build/src folder as <new_file_name>."
    DEPENDS copysrc_to_builddir
)
```

Where `sed -i 's;mermaid::;{mermaid};g'` is the command that replaces the string.

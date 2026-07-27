// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// mdlinklint reports unresolved local Markdown links.
//
// It intentionally has no external dependencies so documentation checks can run
// in a minimal CI image:
//
//	go run tools/mdlinklint/main.go --root dsl_docs --link-root ..
package main

import (
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const maxMarkdownBytes int64 = 4 << 20

type link struct {
	target string
	line   int
	column int
}

type diagnostic struct {
	file    string
	line    int
	column  int
	message string
}

type linter struct {
	root         string
	linkRoot     string
	anchorCache  map[string]map[string]struct{}
	checkedLinks int
}

func main() {
	root := flag.String("root", ".", "directory containing Markdown files to lint")
	linkRoot := flag.String("link-root", "", "boundary for resolved local links (defaults to root)")
	quiet := flag.Bool("quiet", false, "do not print a success summary")
	flag.Parse()

	if flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: mdlinklint [--root directory] [--link-root directory] [--quiet]")
		os.Exit(2)
	}

	canonicalRoot, err := canonicalDirectory(*root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mdlinklint: invalid root: %v\n", err)
		os.Exit(2)
	}
	canonicalLinkRoot := canonicalRoot
	if *linkRoot != "" {
		canonicalLinkRoot, err = canonicalDirectory(*linkRoot)
		if err != nil {
			fmt.Fprintf(os.Stderr, "mdlinklint: invalid link root: %v\n", err)
			os.Exit(2)
		}
	}
	if !pathWithin(canonicalLinkRoot, canonicalRoot) {
		fmt.Fprintf(os.Stderr, "mdlinklint: root %s is outside link root %s\n", canonicalRoot, canonicalLinkRoot)
		os.Exit(2)
	}

	l := linter{
		root:        canonicalRoot,
		linkRoot:    canonicalLinkRoot,
		anchorCache: make(map[string]map[string]struct{}),
	}
	files, err := markdownFiles(canonicalRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "mdlinklint: walk %s: %v\n", *root, err)
		os.Exit(2)
	}

	var diagnostics []diagnostic
	for _, file := range files {
		diagnostics = append(diagnostics, l.lintFile(file)...)
	}
	sort.Slice(diagnostics, func(i, j int) bool {
		if diagnostics[i].file != diagnostics[j].file {
			return diagnostics[i].file < diagnostics[j].file
		}
		if diagnostics[i].line != diagnostics[j].line {
			return diagnostics[i].line < diagnostics[j].line
		}
		return diagnostics[i].column < diagnostics[j].column
	})

	for _, d := range diagnostics {
		fmt.Printf("%s:%d:%d: error: %s\n", d.file, d.line, d.column, d.message)
	}
	if len(diagnostics) != 0 {
		problem := "problem"
		if len(diagnostics) != 1 {
			problem += "s"
		}
		fmt.Printf("mdlinklint: checked %d Markdown files and %d links; found %d %s\n", len(files), l.checkedLinks, len(diagnostics), problem)
		os.Exit(1)
	}
	if !*quiet {
		fmt.Printf("mdlinklint: checked %d Markdown files and %d links; no problems found\n", len(files), l.checkedLinks)
	}
}

func canonicalDirectory(path string) (string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	canonical, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(canonical)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("not a directory: %s", path)
	}
	return filepath.Clean(canonical), nil
}

func pathWithin(root, path string) bool {
	relative, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return relative != ".." && !strings.HasPrefix(relative, ".."+string(os.PathSeparator))
}

func markdownFiles(root string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		if strings.EqualFold(filepath.Ext(entry.Name()), ".md") {
			if entry.Type()&fs.ModeSymlink != 0 {
				return fmt.Errorf("refusing symlinked Markdown source: %s", path)
			}
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

func (l *linter) lintFile(file string) []diagnostic {
	contents, err := readMarkdown(file)
	if err != nil {
		return []diagnostic{l.diagnostic(file, 1, 1, fmt.Sprintf("cannot read file: %v", err))}
	}

	var diagnostics []diagnostic
	inFence := false
	for lineNumber, text := range strings.Split(string(contents), "\n") {
		if isFence(text) {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		for _, candidate := range linksInLine(text) {
			l.checkedLinks++
			if message := l.checkLink(file, candidate.target); message != "" {
				diagnostics = append(diagnostics, l.diagnostic(file, lineNumber+1, candidate.column, message))
			}
		}
	}
	return diagnostics
}

func (l *linter) diagnostic(file string, line, column int, message string) diagnostic {
	relative, err := filepath.Rel(l.root, file)
	if err != nil {
		relative = file
	}
	return diagnostic{file: filepath.ToSlash(relative), line: line, column: column, message: message}
}

func (l *linter) checkLink(source, target string) string {
	if isExternal(target) {
		return ""
	}

	pathPart, fragment, hasFragment := strings.Cut(target, "#")
	pathPart, _, _ = strings.Cut(pathPart, "?")
	decodedPath, err := url.PathUnescape(pathPart)
	if err != nil {
		return fmt.Sprintf("invalid percent-encoding in local link %q", target)
	}
	if filepath.IsAbs(filepath.FromSlash(decodedPath)) {
		return fmt.Sprintf("absolute local link %q is not portable", target)
	}

	resolved := source
	if decodedPath != "" {
		resolved = filepath.Clean(filepath.Join(filepath.Dir(source), filepath.FromSlash(decodedPath)))
	}
	if !pathWithin(l.linkRoot, resolved) {
		return fmt.Sprintf("local link %q escapes link root %s", target, displayPath(l.root, l.linkRoot))
	}
	canonical, err := filepath.EvalSymlinks(resolved)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Sprintf("unresolved local link %q (resolved to %s)", target, displayPath(l.root, resolved))
		}
		return fmt.Sprintf("cannot resolve local link %q: %v", target, err)
	}
	if !pathWithin(l.linkRoot, canonical) {
		return fmt.Sprintf("local link %q escapes link root through a symbolic link", target)
	}
	info, err := os.Stat(canonical)
	if err != nil {
		return fmt.Sprintf("cannot inspect local link %q: %v", target, err)
	}

	if hasFragment && fragment != "" && strings.EqualFold(filepath.Ext(resolved), ".md") {
		decodedFragment, err := url.PathUnescape(fragment)
		if err != nil {
			return fmt.Sprintf("invalid percent-encoding in fragment %q", target)
		}
		if !info.Mode().IsRegular() {
			return fmt.Sprintf("Markdown fragment target %q is not a regular file", target)
		}
		anchors, err := l.anchors(canonical)
		if err != nil {
			return fmt.Sprintf("cannot read link target %q: %v", target, err)
		}
		if _, ok := anchors[decodedFragment]; !ok {
			return fmt.Sprintf("unresolved fragment #%s in local link %q", decodedFragment, target)
		}
	}
	return ""
}

func displayPath(root, path string) string {
	relative, err := filepath.Rel(root, path)
	if err != nil {
		return filepath.ToSlash(path)
	}
	return filepath.ToSlash(relative)
}

func (l *linter) anchors(file string) (map[string]struct{}, error) {
	if anchors, ok := l.anchorCache[file]; ok {
		return anchors, nil
	}
	contents, err := readMarkdown(file)
	if err != nil {
		return nil, err
	}

	anchors := make(map[string]struct{})
	counts := make(map[string]int)
	for _, line := range strings.Split(string(contents), "\n") {
		heading, ok := headingText(line)
		if !ok {
			continue
		}
		anchor := githubAnchor(heading)
		if anchor == "" {
			continue
		}
		count := counts[anchor]
		counts[anchor]++
		if count != 0 {
			anchor = fmt.Sprintf("%s-%d", anchor, count)
		}
		anchors[anchor] = struct{}{}
	}
	l.anchorCache[file] = anchors
	return anchors, nil
}

func readMarkdown(file string) ([]byte, error) {
	info, err := os.Stat(file)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("not a regular file")
	}
	if info.Size() > maxMarkdownBytes {
		return nil, fmt.Errorf("file is %d bytes; limit is %d", info.Size(), maxMarkdownBytes)
	}
	return os.ReadFile(file)
}

func isExternal(target string) bool {
	lower := strings.ToLower(strings.TrimSpace(target))
	if strings.HasPrefix(lower, "//") {
		return true
	}
	for _, scheme := range []string{"http:", "https:", "mailto:", "tel:", "data:"} {
		if strings.HasPrefix(lower, scheme) {
			return true
		}
	}
	return false
}

func isFence(line string) bool {
	trimmed := strings.TrimLeft(line, " \t")
	return strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~")
}

func headingText(line string) (string, bool) {
	trimmed := strings.TrimLeft(line, " \t")
	if !strings.HasPrefix(trimmed, "#") {
		return "", false
	}
	level := 0
	for level < len(trimmed) && trimmed[level] == '#' {
		level++
	}
	if level == len(trimmed) || (trimmed[level] != ' ' && trimmed[level] != '\t') {
		return "", false
	}
	return strings.TrimSpace(strings.TrimRight(strings.TrimSpace(trimmed[level:]), "#")), true
}

func githubAnchor(heading string) string {
	var out strings.Builder
	dash := false
	for _, r := range strings.ToLower(heading) {
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '_', r == '-':
			out.WriteRune(r)
			dash = false
		case unicode.IsSpace(r):
			if out.Len() != 0 && !dash {
				out.WriteByte('-')
				dash = true
			}
		}
	}
	return strings.Trim(out.String(), "-")
}

func linksInLine(line string) []link {
	var links []link
	codeTicks := false
	for index := 0; index < len(line); index++ {
		if line[index] == '`' && !escaped(line, index) {
			codeTicks = !codeTicks
			continue
		}
		if codeTicks || line[index] != '[' || escaped(line, index) {
			continue
		}

		labelEnd := findClosingBracket(line, index+1)
		if labelEnd == -1 || labelEnd+1 >= len(line) || line[labelEnd+1] != '(' {
			continue
		}
		targetStart := labelEnd + 2
		targetEnd := findClosingParen(line, targetStart)
		if targetEnd == -1 {
			continue
		}
		target, offset, ok := destination(line[targetStart:targetEnd])
		if ok {
			links = append(links, link{target: target, column: utf8.RuneCountInString(line[:targetStart+offset]) + 1})
		}
		index = targetEnd
	}
	return links
}

func escaped(text string, index int) bool {
	backslashes := 0
	for index > 0 && text[index-1] == '\\' {
		backslashes++
		index--
	}
	return backslashes%2 != 0
}

func findClosingBracket(text string, start int) int {
	for index := start; index < len(text); index++ {
		if text[index] == ']' && !escaped(text, index) {
			return index
		}
	}
	return -1
}

func findClosingParen(text string, start int) int {
	depth := 0
	for index := start; index < len(text); index++ {
		switch text[index] {
		case '(':
			if !escaped(text, index) {
				depth++
			}
		case ')':
			if escaped(text, index) {
				continue
			}
			if depth == 0 {
				return index
			}
			depth--
		}
	}
	return -1
}

func destination(raw string) (target string, offset int, ok bool) {
	trimmedLeft := strings.TrimLeft(raw, " \t")
	offset = len(raw) - len(trimmedLeft)
	if trimmedLeft == "" {
		return "", offset, false
	}
	if trimmedLeft[0] == '<' {
		end := strings.IndexByte(trimmedLeft, '>')
		if end == -1 {
			return "", offset, false
		}
		return trimmedLeft[1:end], offset + 1, true
	}
	for end, r := range trimmedLeft {
		if unicode.IsSpace(r) {
			return trimmedLeft[:end], offset, true
		}
	}
	return trimmedLeft, offset, true
}

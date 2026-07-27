// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLinksInLine(t *testing.T) {
	links := linksInLine("[good](one.md) and [`also good`](two.md#section) and `[not a link](no.md)`")
	if len(links) != 2 {
		t.Fatalf("found %d links, want 2", len(links))
	}
	if links[0].target != "one.md" || links[1].target != "two.md#section" {
		t.Fatalf("unexpected targets: %#v", links)
	}
}

func TestLinterResolvesPathsAndFragments(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "guide.md"), "# Guide title\n\n## Repeated Heading\n\n## Repeated Heading\n")
	writeFile(t, filepath.Join(root, "index.md"), strings.Join([]string{
		"[guide](guide.md#guide-title)",
		"[duplicate](guide.md#repeated-heading-1)",
		"[directory](assets/)",
		"[remote](https://example.com/docs)",
		"[anchor](#local-anchor)",
		"",
		"# Local anchor",
	}, "\n"))
	if err := os.Mkdir(filepath.Join(root, "assets"), 0o755); err != nil {
		t.Fatal(err)
	}

	l := testLinter(root, root)
	if diagnostics := l.lintFile(filepath.Join(root, "index.md")); len(diagnostics) != 0 {
		t.Fatalf("unexpected diagnostics: %#v", diagnostics)
	}
	if l.checkedLinks != 5 {
		t.Fatalf("checked %d links, want 5", l.checkedLinks)
	}
}

func TestLinterReportsUnresolvedPathAndFragment(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "index.md"), "[missing](missing.md)\n[fragment](guide.md#missing)\n")
	writeFile(t, filepath.Join(root, "guide.md"), "# Present\n")

	l := testLinter(root, root)
	diagnostics := l.lintFile(filepath.Join(root, "index.md"))
	if len(diagnostics) != 2 {
		t.Fatalf("found %d diagnostics, want 2: %#v", len(diagnostics), diagnostics)
	}
	if !strings.Contains(diagnostics[0].message, "unresolved local link") {
		t.Fatalf("unexpected path diagnostic: %q", diagnostics[0].message)
	}
	if !strings.Contains(diagnostics[1].message, "unresolved fragment") {
		t.Fatalf("unexpected fragment diagnostic: %q", diagnostics[1].message)
	}
}

func TestLinterEnforcesLinkRoot(t *testing.T) {
	workspace := t.TempDir()
	docs := filepath.Join(workspace, "docs")
	if err := os.Mkdir(docs, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(workspace, "guide.md"), "# Guide\n")
	writeFile(t, filepath.Join(docs, "index.md"), "[guide](../guide.md#guide)\n")

	t.Run("rejects target outside boundary", func(t *testing.T) {
		l := testLinter(docs, docs)
		diagnostics := l.lintFile(filepath.Join(docs, "index.md"))
		if len(diagnostics) != 1 || !strings.Contains(diagnostics[0].message, "escapes link root") {
			t.Fatalf("unexpected diagnostics: %#v", diagnostics)
		}
	})

	t.Run("allows target outside scan root within boundary", func(t *testing.T) {
		l := testLinter(docs, workspace)
		if diagnostics := l.lintFile(filepath.Join(docs, "index.md")); len(diagnostics) != 0 {
			t.Fatalf("unexpected diagnostics: %#v", diagnostics)
		}
	})
}

func TestLinterRejectsSymlinkEscape(t *testing.T) {
	workspace := t.TempDir()
	docs := filepath.Join(workspace, "docs")
	if err := os.Mkdir(docs, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(workspace, "outside.md"), "# Outside\n")
	if err := os.Symlink(filepath.Join(workspace, "outside.md"), filepath.Join(docs, "escape.md")); err != nil {
		t.Skipf("cannot create symlink: %v", err)
	}
	writeFile(t, filepath.Join(docs, "index.md"), "[escape](escape.md#outside)\n")

	l := testLinter(docs, docs)
	diagnostics := l.lintFile(filepath.Join(docs, "index.md"))
	if len(diagnostics) != 1 || !strings.Contains(diagnostics[0].message, "through a symbolic link") {
		t.Fatalf("unexpected diagnostics: %#v", diagnostics)
	}
}

func TestLinterRejectsOversizedMarkdown(t *testing.T) {
	root := t.TempDir()
	large := filepath.Join(root, "large.md")
	writeFile(t, large, "")
	if err := os.Truncate(large, maxMarkdownBytes+1); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(root, "index.md"), "[large](large.md#heading)\n")

	l := testLinter(root, root)
	diagnostics := l.lintFile(filepath.Join(root, "index.md"))
	if len(diagnostics) != 1 || !strings.Contains(diagnostics[0].message, "limit is") {
		t.Fatalf("unexpected diagnostics: %#v", diagnostics)
	}
}

func testLinter(root, linkRoot string) linter {
	return linter{
		root:        root,
		linkRoot:    linkRoot,
		anchorCache: make(map[string]map[string]struct{}),
	}
}

func writeFile(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

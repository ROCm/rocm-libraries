#!/usr/bin/env bash
# .claude/setup.sh — 开发环境配置 + rebase 工作流
#
# 功能：
#   1. 配置 Claude Code memory 软链接（首次运行）
#   2. 将 rebase 分支 rebase 到 upstream 最新
#   3. 列出 zhewan/* 分支，让用户选择
#   4. 将选中的分支 rebase 到 rebase 分支上
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLAUDE_DIR="$REPO_ROOT/.claude"
cd "$REPO_ROOT"

REBASE_BRANCH="zhewan/ck/rebase"
UPSTREAM="origin/develop"

# ============================================================
# 1. Claude Code memory 软链接配置
# ============================================================
setup_memory() {
    PROJECT_SLUG="$(echo "$REPO_ROOT" | sed 's|^/||; s|/|-|g')"
    SYSTEM_PROJECT_DIR="$HOME/.claude/projects/-${PROJECT_SLUG}"
    mkdir -p "$SYSTEM_PROJECT_DIR"

    SYSTEM_MEMORY="$SYSTEM_PROJECT_DIR/memory"
    PROJECT_MEMORY="$CLAUDE_DIR/memory"

    if [ -L "$SYSTEM_MEMORY" ]; then
        CURRENT_TARGET="$(readlink "$SYSTEM_MEMORY")"
        if [ "$CURRENT_TARGET" = "$PROJECT_MEMORY" ]; then
            echo "[OK] memory 软链接已就绪"
        else
            rm "$SYSTEM_MEMORY"
            ln -s "$PROJECT_MEMORY" "$SYSTEM_MEMORY"
            echo "[OK] memory 软链接已修正"
        fi
    elif [ -d "$SYSTEM_MEMORY" ]; then
        if [ -z "$(ls -A "$SYSTEM_MEMORY")" ]; then
            rmdir "$SYSTEM_MEMORY"
        else
            cp -n "$SYSTEM_MEMORY"/* "$PROJECT_MEMORY"/ 2>/dev/null || true
            rm -rf "$SYSTEM_MEMORY"
        fi
        ln -s "$PROJECT_MEMORY" "$SYSTEM_MEMORY"
        echo "[OK] memory 软链接已创建"
    else
        ln -s "$PROJECT_MEMORY" "$SYSTEM_MEMORY"
        echo "[OK] memory 软链接已创建"
    fi
}

# ============================================================
# 2. Rebase 工作流
# ============================================================
rebase_workflow() {
    # 检查工作区是否干净
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "[ERROR] 工作区有未提交的修改，请先 commit 或 stash"
        exit 1
    fi

    echo ""
    echo "=== 同步 upstream ==="
    git fetch origin
    echo "[OK] fetch 完成"

    # 切到 rebase 分支并 rebase upstream
    echo ""
    echo "=== Rebase $REBASE_BRANCH onto $UPSTREAM ==="
    git checkout "$REBASE_BRANCH"
    if git rebase "$UPSTREAM"; then
        echo "[OK] $REBASE_BRANCH 已 rebase 到 $UPSTREAM 最新"
    else
        echo "[ERROR] rebase 冲突，请手动解决后运行 git rebase --continue"
        exit 1
    fi

    # 收集 zhewan/* 分支（本地 + 远端去重），排除 rebase 分支自身
    echo ""
    echo "=== 可选分支 ==="
    local branches=()
    while IFS= read -r b; do
        [ -n "$b" ] && branches+=("$b")
    done < <(
        {
            git branch --list 'zhewan/*' --format='%(refname:short)'
            git branch -r --list 'origin/zhewan/*' --format='%(refname:short)' | sed 's|^origin/||'
        } | sort -u | grep -v "^${REBASE_BRANCH}$"
    )

    if [ ${#branches[@]} -eq 0 ]; then
        echo "没有找到其他 zhewan/* 分支"
        echo "当前已在 $REBASE_BRANCH（已 rebase 到最新）"
        return
    fi

    local i=1
    for b in "${branches[@]}"; do
        echo "  $i) $b"
        ((i++))
    done
    echo "  0) 不切换，留在 $REBASE_BRANCH"
    echo ""

    read -rp "选择分支编号: " choice

    if [ "$choice" = "0" ] || [ -z "$choice" ]; then
        echo "留在 $REBASE_BRANCH"
        return
    fi

    if [ "$choice" -lt 1 ] || [ "$choice" -gt ${#branches[@]} ] 2>/dev/null; then
        echo "[ERROR] 无效选择"
        exit 1
    fi

    local target="${branches[$((choice - 1))]}"
    echo ""
    echo "=== 切换到 $target 并 rebase onto $REBASE_BRANCH ==="

    # 如果本地没有这个分支，从远端 checkout
    if ! git show-ref --verify --quiet "refs/heads/$target"; then
        git checkout -b "$target" "origin/$target"
    else
        git checkout "$target"
    fi

    if git rebase "$REBASE_BRANCH"; then
        echo "[OK] $target 已 rebase 到 $REBASE_BRANCH 最新内容"
    else
        echo "[ERROR] rebase 冲突，请手动解决后运行 git rebase --continue"
        exit 1
    fi

    echo ""
    echo "=== 完成 ==="
    echo "  当前分支: $(git branch --show-current)"
    echo "  基于: $REBASE_BRANCH (已同步 $UPSTREAM)"
}

# ============================================================
# main
# ============================================================
echo "=== Claude Code 开发环境配置 ==="
echo "项目目录: $REPO_ROOT"
echo ""

setup_memory
rebase_workflow

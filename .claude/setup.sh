#!/usr/bin/env bash
# .claude/setup.sh — 在新机器上 clone 仓库后运行一次，自动配置 Claude Code 开发环境
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLAUDE_DIR="$REPO_ROOT/.claude"

# Claude Code 用绝对路径编码项目目录：/home/user/foo → -home-user-foo
PROJECT_SLUG="$(echo "$REPO_ROOT" | sed 's|^/||; s|/|-|g')"
SYSTEM_PROJECT_DIR="$HOME/.claude/projects/-${PROJECT_SLUG}"

echo "=== Claude Code 开发环境配置 ==="
echo "项目目录: $REPO_ROOT"
echo "系统项目目录: $SYSTEM_PROJECT_DIR"
echo ""

# 1. 确保系统项目目录存在
mkdir -p "$SYSTEM_PROJECT_DIR"

# 2. 软链接 memory 目录
SYSTEM_MEMORY="$SYSTEM_PROJECT_DIR/memory"
PROJECT_MEMORY="$CLAUDE_DIR/memory"

if [ -L "$SYSTEM_MEMORY" ]; then
    CURRENT_TARGET="$(readlink "$SYSTEM_MEMORY")"
    if [ "$CURRENT_TARGET" = "$PROJECT_MEMORY" ]; then
        echo "[OK] memory 软链接已正确指向项目目录"
    else
        echo "[FIX] memory 软链接指向错误 ($CURRENT_TARGET)，修正中..."
        rm "$SYSTEM_MEMORY"
        ln -s "$PROJECT_MEMORY" "$SYSTEM_MEMORY"
        echo "[OK] memory 软链接已修正"
    fi
elif [ -d "$SYSTEM_MEMORY" ]; then
    if [ -z "$(ls -A "$SYSTEM_MEMORY")" ]; then
        rmdir "$SYSTEM_MEMORY"
        ln -s "$PROJECT_MEMORY" "$SYSTEM_MEMORY"
        echo "[OK] memory 空目录已替换为软链接"
    else
        echo "[WARN] $SYSTEM_MEMORY 非空，合并现有 memory 文件..."
        cp -n "$SYSTEM_MEMORY"/* "$PROJECT_MEMORY"/ 2>/dev/null || true
        rm -rf "$SYSTEM_MEMORY"
        ln -s "$PROJECT_MEMORY" "$SYSTEM_MEMORY"
        echo "[OK] memory 已合并并替换为软链接"
    fi
else
    ln -s "$PROJECT_MEMORY" "$SYSTEM_MEMORY"
    echo "[OK] memory 软链接已创建"
fi

echo ""
echo "=== 配置完成 ==="
echo "  CLAUDE.md:     $CLAUDE_DIR/CLAUDE.md"
echo "  settings.json: $CLAUDE_DIR/settings.json"
echo "  memory:        $SYSTEM_MEMORY -> $PROJECT_MEMORY"
